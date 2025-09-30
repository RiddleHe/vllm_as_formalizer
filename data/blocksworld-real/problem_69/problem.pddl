(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        green - block
        purple - block
        orange - block
        red - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable purple)
        (clear purple)
        (ontable orange)
        (clear orange)
        (ontable red)
        (on blue red)
        (clear blue)
        ; yellow
        ; green
        ; purple
        ; orange
        ; red -> blue
    )
    (:goal (and (on yellow orange)
        (on red blue)
        (on purple red)
        (on green purple)))
)
    