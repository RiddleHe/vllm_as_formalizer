(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        orange - block
        green - block
        red - block
        blue - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on orange yellow)
        (clear orange)
        (ontable green)
        (clear green)
        (ontable red)
        (on blue red)
        (on purple blue)
        (clear purple)
        ; yellow -> orange
        ; green
        ; red -> blue -> purple
    )
    (:goal (and (on purple blue)
        (on green purple)
        (on orange green)
        (on yellow orange)
        (on red yellow)))
)
    