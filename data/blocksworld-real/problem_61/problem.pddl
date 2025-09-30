(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        purple - block
        red - block
        green - block
        orange - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on purple blue)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        ; blue -> purple
        ; red
        ; green
        ; orange
        ; yellow
    )
    (:goal (and (on yellow orange)
        (on blue red)
        (on green purple)))
)
    