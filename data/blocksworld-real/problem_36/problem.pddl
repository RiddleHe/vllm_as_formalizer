(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        green - block
        red - block
        blue - block
        yellow - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable green)
        (clear green)
        (ontable red)
        (on blue red)
        (clear blue)
        (ontable yellow)
        (on orange yellow)
        (clear orange)
        ; purple
        ; green
        ; red -> blue
        ; yellow -> orange
    )
    (:goal (and (on yellow blue)
        (on red yellow)
        (on green purple)
        (on orange green)))
)
    